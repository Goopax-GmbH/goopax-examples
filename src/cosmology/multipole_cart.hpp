#pragma once
#include "zero.hpp"


template <typename T> T factorial(T x)
{
  assert(x >= 0);
  if (x == 0)
    {
      return 1;
    }
  else
    {
      return x * factorial(x-1);
    }
}

template <typename T> T factorial2(T x)
{
  assert(x >= -1);
  if (x == -1)
    {
      return 1;
    }
  else
    {
      return x * factorial2(x-2);
    }
}

template<typename... T>
struct multipole;

template<>
struct multipole<>
{
  static constexpr int order = -1;
  template <int N>
  using get_t = void;
  
  static multipole zero()
  {
    return {};
  }
  template <typename shift_t, typename mass_t>
  static multipole from_particle(Vector<shift_t, 3>, mass_t)
  {
    return {};
  }
  multipole& operator+=(const multipole&)
  {
    return *this;
  }
  template <typename scale_t>
  multipole scale_ext(scale_t) const
  {
    return {};
  }
  template <typename scale_t>
  multipole scale_loc(scale_t) const
  {
    return {};
  }
  multipole rot(Tint = 1) const
  {
    return {};
  }
  template <typename shift_t>
  multipole shift_ext(Vector<shift_t, 3>) const
  {
    return {};
  }
  template <typename shift_t>
  multipole shift_loc(Vector<shift_t, 3>) const
  {
    return {};
  }
};


template <typename DEST>
void copy_all(DEST)
{
}

template <typename DEST, typename A, typename... B>
void copy_all(DEST&& d, A a, B... b)
{
  std::ranges::copy(a, d);
  copy_all(d + a.size(), b...);
}

template<typename T, typename... TPREV>
struct
alignas(sizeof...(TPREV)==4 ? 16 : 4)
multipole<T, TPREV...> : public multipole<TPREV...>
{
  using prev_t = multipole<TPREV...>;
  static constexpr int order = prev_t::order + 1;

  template <int N>
  using get_t = typename conditional<N==order, multipole, typename prev_t::template get_t<N>>::type;
  
  using value_type = T;

  prev_t& prev()
  {
    return *this;
  }
  const prev_t& prev() const
  {
    return *this;
  }

  template <int N> get_t<N>& get()
  {
    return *this;
  }
  template <int N> const get_t<N>& get() const
  {
    return *this;
  }
  
  Tuint get_index(array<Tint, order> indices) const
  {
    Tuint index = 0;
    for (uint i = 0; i < order; ++i)
      {
	index = index * 3 + indices[i];
      }
    return index;
  }

  //template<int K=order, typename Tuse = value_type>
  const T coeff(array<Tint, order> indices) const
  {
    return A[indexmap[get_index(indices)]];
  }

  template <typename... I>
  value_type coeff(I&&... indices) const
  {
    array<Tint, (indices.i.size() + ...)> combined;
    copy_all(combined.begin(), (indices.i)...);
    //return this->coeff<order>(combined);
    return A[indexmap[get_index(combined)]];
  }
  
  Vector<T, (order + 1) * (order + 2) / 2> A;

  struct index_t
  {
    Tint pos;
    array<Tint, 3> e;
    array<Tint, order> i;
    Tint factor;

    template <typename U>
    U prod(const Vector<U, 3>& v) const
    {
      return pow(v[0], e[0]) * pow(v[1], e[1]) * pow(v[2], e[2]);
    }

    template<int K, typename FUNC>
    auto cycle(FUNC func) const
    {
      using ret_t = decltype(func(get_t<K>::indices[0], get_t<order-K>::indices[0]));
      static_assert(goopax_is_floating_point<typename make_cpu<ret_t>::type>::value);
      
      array<Tbool, order> p;
      fill(p.begin(), p.begin() + K, false);
      fill(p.begin() + K, p.end(), true);
      map<Tint, pair<Tuint, ret_t>> cache;

      do
	{
	  array<Tint, K> q;
	  array<Tint, order - K> w;
	  Tuint e_hash = 0;
	  auto pq = q.begin();
	  auto pw = w.begin();
	  for (int k = 0; k < order; ++k)
	    {
	      e_hash += (1u << 8 * i[k]);
	      if (p[k])
		*pw++ = i[k];
	      else
		*pq++ = i[k];
	    }
	  {
	    uint check = 0;
	    for (auto& qq : get_t<K>::indices)
	      {
		for (auto& ww : get_t<order-K>::indices)
		  {
		    if (qq.i == q && ww.i == w)
		      {
			if (!cache.contains(qq.pos))
			  {
			    cache[qq.pos] = {0.0, func(qq, ww)};
			  }
			++cache[qq.pos].first;
			++check;
		      }
		  }
	      }
	    assert(check == 1);
	  }
	} while (ranges::next_permutation(p).found);

      ret_t ret = 0;
      for (auto& q : cache)
	{
	  ret += static_cast<ret_t>(q.second.first) * static_cast<ret_t>(q.second.second);
	}
      return ret;
    }

    friend ostream& operator<<(ostream& s, const index_t& i)
    {
      return s << "[pos=" << i.pos << ", e=" << i.e << ", i=" << i.i << ", factor=" << i.factor << "]";
    }
  };

  T& operator[](index_t i)
  {
    return A[i.pos];
  }
  const T& operator[](index_t i) const
  {
    return A[i.pos];
  }
  
  template <typename... I>
  value_type operator ()(I&&... indices) const
  {
    return coeff(indices...);
  }

  template <typename... I>
  value_type& operator ()(I&&... indices)
  {
    return coeff(indices...);
  }

  
  
  static array<Tint, pow<order, 1>(3)> indexmap;
  static const array<index_t, (order + 1) * (order + 2) / 2> indices;

  multipole rot(Tint step = 1) const
  {
    if (step < 0)
      return rot(step + 3);
    if (step == 0)
      return *this;

    multipole ret;
    ret.prev() = prev().rot();

    for (auto& i : indices)
      {
	array<Tint, order> io = i.i;
	for (auto& o : io)
	  {
	    o = (o+1)%3;
	  }
	ret[i] = this->coeff(io);
      }
    return ret.rot(step-1);
  }

  multipole() = default;

  multipole& operator+=(const multipole& b)
  {
    prev() += b.prev();
    A += b.A;
    return *this;
  }

private:
  template<typename FUNC>
  static auto sum(FUNC func)
  {
    decltype(func(0)) ret = 0;
    for (int k = 0; k < 3; ++k)
      {
	ret += func(k);
      }
    return ret;
  }

public:
  template<typename FUNC>
  static auto indexsum(FUNC func)
  {
    decltype(func(indices[0])) ret = 0;
    for (auto& i : indices)
      {
	ret += func(i) * i.factor;
      }
    return ret;
  }

  template <typename scale_t>
  multipole scale_ext(scale_t s) const
  {
    multipole ret;
    ret.prev() = prev().scale_ext(s);
    ret.A = A * static_cast<T>(pow<order, 1>(s));
    return ret;
  }
 
  template <typename scale_t>
  multipole scale_loc(scale_t s) const
  {
    multipole ret;
    ret.prev() = prev().scale_loc(s);
    ret.A = A * static_cast<T>(pow<-order-1, 1>(s));
    return ret;
  }

  
  template<typename MP, typename shift_t>
  void shift_ext_contrib(Vector<shift_t, 3> a, const MP& M)
  {
    for (auto& i : indices)
      {
	const double factortab[] = {0, 1, 3.0/2, 10.0/4, 35.0/8};

	static_assert(MP::order < order);
	
	if constexpr (order >= 1)
	  {
	    double factor;
	    if (MP::order == 0)
	      factor = factortab[order];
	    if (MP::order == 1)
	      factor = factortab[order];
	    if (MP::order == 2)
	      factor = factortab[order]*2.0/3;
	    if (MP::order == 3)
	      factor = 7.0 / 8;
	    
	    (*this)[i] +=
	      static_cast<T>(factor) * i.template cycle<MP::order>([&](auto q, auto w) { return static_cast<T>(w.prod(a)) * M[q]; });
	  }



	if constexpr (order >= 2)
	  {
	    if constexpr(MP::order == 0)
	      {
		const double factortab[] = {0,0, -1.0/2, -1.0/2, -5.0/8};
		(*this)[i] +=
		  + static_cast<T>(static_cast<shift_t>(factortab[order]) * a.squaredNorm() * i.template cycle<order-2>([&](auto q, auto w){return q.prod(a) * (w.i[0]==w.i[1]);})) * M.coeff({}) ;
	      }
	    if constexpr(MP::order == 1)
	      {
		double factor = 1;
		if (order == 4)
		  {
		    factor = 5.0/4;
		  }
		(*this)[i] -=
		  static_cast<T>(static_cast<shift_t>(factor) * i.template cycle<order-2>([&](auto q, auto w){return q.prod(a) * (w.i[0]==w.i[1]);})) * get_t<1>::indexsum([&](auto j) { return M[j] * static_cast<T>(j.prod(a)); }) ;
	      }
	    if constexpr(MP::order == 2)
	      {
		const double factortab[] = {0,0, 0, 2.0/3, 5.0/6};
		(*this)[i] += 
		  static_cast<T>(-factortab[order]) * get_t<1>::indexsum([&](auto j) {return static_cast<T>(j.prod(a)) * i.template cycle<1>([&](auto q, auto w) {return M.coeff(j, q) * static_cast<T>(w.template cycle<2>([&](auto wa, auto wb){ return wb.prod(a) * (wa.i[0] == wa.i[1]);}));});});
		  
	      }
	  }

	if constexpr (order == 3)
	  {
	    if constexpr(MP::order == 1)
	      {
		(*this)[i] +=
		  - static_cast<T>(0.5) * i.template cycle<1>([&](auto q, auto w){return M[q] * static_cast<T>((a.squaredNorm() * (w.i[0]==w.i[1])));});
	      }
	  }
	if constexpr (order == 4)
	  {
	    if constexpr(MP::order == 0)
	      {
		(*this)[i] +=
		  + static_cast<T>(static_cast<shift_t>(1.0 / 16 * i.template cycle<2>([&](auto q, auto w) -> double {return (q.i[0] == q.i[1] && w.i[0] == w.i[1]);})) * pow2(a.squaredNorm())) * M.coeff({});
	      }
	    if constexpr(MP::order == 1)
	      {
		(*this)[i] +=
		  - static_cast<T>(static_cast<shift_t>(5.0 / 8) * a.squaredNorm()) * i.template cycle<1>([&](auto q, auto w) {return M(q) * static_cast<T>(w.template cycle<1>([&](auto wa, auto wb){return a[wa.i[0]] * (wb.i[0] == wb.i[1]);}));})
		  + static_cast<T>(static_cast<shift_t>(1.0/4 * i.template cycle<2>([&](auto q, auto w) -> double{ return (q.i[0] == q.i[1] && w.i[0] == w.i[1]); })) * a.squaredNorm()) * get_t<1>::indexsum([&](auto j) { return M[j] * static_cast<T>(j.prod(a)); });
	      }
	    if constexpr(MP::order == 2)
	      {
		(*this)[i] +=
		  - static_cast<T>(static_cast<shift_t>(5.0 / 12) * a.squaredNorm()) * i.template cycle<2>([&](auto q, auto w) {return M(q) * ((w.i[0] == w.i[1]));})
		  + static_cast<T>(1.0 / 6 * i.template cycle<2>([&](auto q, auto w) {return static_cast<double>(q.i[0] == q.i[1] && w.i[0] == w.i[1]);})) * get_t<2>::indexsum([&](auto j) {return M.coeff(j) * static_cast<T>(j.prod(a));});
	      }
	    if constexpr(MP::order == 3)
	      {
		(*this)[i] +=
		  - static_cast<T>(1.0 / 2) * get_t<1>::indexsum([&](auto j) {return static_cast<T>(j.prod(a)) * i.template cycle<2>([&](auto q, auto w) {return M.coeff(j, q) * (w.i[0] == w.i[1]);});});
	      }
	  }
      }
    if constexpr (MP::order > 0)
      {
	shift_ext_contrib(a, M.prev());
      }
    if constexpr (M.order == order-1 && order > 0)
      {
	prev().template shift_ext_contrib(a, M.prev());
      }
  }

  template <typename shift_t>
  multipole shift_ext(Vector<shift_t, 3> a) const
  {
    multipole M = *this;
    M.shift_ext_contrib(a, prev());
    return M;
  }

  template <typename shift_t>
  multipole shift_loc(Vector<shift_t, 3> a) const
  {
    multipole M;

    M.prev() = prev().shift_loc(a);
    M.A = A;
    a = -a;

    if constexpr (order >= 1)
      {
	for (auto& i : M.get<0>().indices)
	  {
	    M.get<0>()[i] += get<order>().indexsum([&](auto j){return coeff(i, j) * static_cast<T>(j.prod(a));});
	  }
      }

    if constexpr (order >= 2)
      {
	for (auto& i : M.get<1>().indices)
	  {
	    M.get<1>()[i] += order * get<order-1>().indexsum([&](auto j){return coeff(i, j) * static_cast<T>(j.prod(a));});
	  }
      }
    if constexpr (order >= 3)
      {
	for (auto& i : M.get<2>().indices)
	  {
	    M.get<2>()[i] += 3*(order-2)*get<order-2>().indexsum([&](auto j){return coeff(i, j) * static_cast<T>(j.prod(a));});
	  }
      }
    if constexpr(order == 4)
      {
	for (auto& i : M.get<3>().indices)
	  {
	    M.get<3>()[i] += 4*get<order-3>().indexsum([&](auto j){return coeff(i, j) * static_cast<T>(j.prod(a));});
	  }	
      }
    return M;
  }
  

  template<typename MAX, typename MP, typename shift_t>
  void makelocal_add_contrib(Vector<shift_t, 3> a, const MP& M)
  {
    if constexpr(order + MP::order <= MAX::order)
      {
	using Tuse = typename MAX::get_t<order + MP::order>::value_type ;
	shift_t inva = pow<-1, 2>(a.squaredNorm());
	const Vector<shift_t, 3> e = (a * pow2(inva));

	for (auto& i : indices)
	  {
	    Tuse contrib = 0;
	    
	    contrib +=
	      static_cast<Tuse>(factorial2(2*(MP::order + order)-1) / factorial2(static_cast<Tdouble>(2*MP::order-1)) * pow(-1, order) / factorial(order)
				* inva * i.prod(e)) * M.indexsum([&](auto j){return static_cast<Tuse>(M[j]) * static_cast<T>(j.prod(e));});

	
	    if constexpr(MP::order >= 1 && order >= 1)
	      {
		constexpr array<array<double, 4>, 4> factortab =
		  {
		    {{0,  0,      0, 0},
		     {0,  1,      2, 3},
		     {0, -3.0/2, -5, 0},
		     {0,  5.0/2,  0, 0}}
		  };
		
		contrib +=
		  static_cast<T>(factortab[order][MP::order] * pow3(inva)) * MAX::template get_t<MP::order-1>::indexsum([&](auto j) {return static_cast<T>(j.prod(e)) * i.template cycle<1>([&](auto q, auto w){return M.template coeff(j, q) * static_cast<T>(w.prod(e));});});
	      }
		
	    if constexpr (order == 2)
	      {
		if constexpr (MP::order == 2)
		  {
		    contrib += static_cast<T>(pow5(inva)) * static_cast<Tuse>(M[i]);
		  }
	      }
	    if constexpr (order >= 2)
	      {
		double factor = (MP::order == 1 ? 5.0 : 1.0) / 2.0;
		if (order == 2)
		  {
		    factor = -(MP::order*2.0+1) / 2;
		  }
		if (order == 4)
		  {
		    factor = -5.0/8;
		  }
		
		contrib += static_cast<T>(factor * pow3(inva) * i.template cycle<order-2>([&](auto q, auto w) { return q.prod(e) * (w.i[0] == w.i[1]); })) * M.indexsum([&](auto j) { return M[j] * static_cast<T>(j.prod(e)); });
	      }
	    if constexpr (order == 3)
	      {
		if constexpr (MP::order == 1)
		  {
		    contrib +=
		      static_cast<Tuse>(-1.0 / 2 * pow5(inva)) * i.template cycle<1>([&](auto q, auto w) {return M.template coeff(q) * (w.i[0] == w.i[1]);});
		  }
	      }
	    if constexpr (order == 4)
	      {
		if constexpr (MP::order == 0)
		  {
		    contrib +=
		      + static_cast<Tuse>(1.0 / 16 * pow5(inva) * i.template cycle<2>([&](auto q, auto w) {return static_cast<shift_t>(q.i[0] == q.i[1] && w.i[0] == w.i[1]);})) * static_cast<Tuse>(M.template coeff({}));
		  }
	      }
	    (*this)[i] += contrib;
	  }
      }
    if constexpr (MP::order > 0)
      {
	makelocal_add_contrib<MAX>(a, M.prev());
      }
    if constexpr (M.order == MAX::order && order > 0)
      {
	prev().template makelocal_add_contrib<MAX>(a, M);
      }
  }

  template <typename shift_t>
  multipole makelocal(Vector<shift_t, 3> a) const
  {
    a = -a;
    multipole M = multipole::zero();
    M.makelocal_add_contrib<multipole>(a, *this);

    return M;
  }

  template <typename shift_t, typename mass_t>
  static multipole from_particle(Vector<shift_t, 3> a, mass_t mass)
  {
    auto tmp = multipole::zero();
    tmp.template get<0>().A[0] = static_cast<typename get_t<0>::value_type>(-mass);
    auto M = tmp.shift_ext((-a).eval());
    return M;
  }

  static multipole zero()
  {
    multipole ret;
    static_cast<prev_t&>(ret) = prev_t::zero();
    ret.A.fill(0);
    return ret;
  }

#if CALC_POTENTIAL
  template <typename shift_t>
  typename get_t<0>::value_type calc_loc_potential(Vector<shift_t, 3> r) const
  {
    T contrib = 0;
    for (auto& i : indices)
      {
	contrib += (*this)[i] * i.factor * (pow(-r[0], i.e[0]) * pow(-r[1], i.e[1]) * pow(-r[2], i.e[2]));
      }

    typename get_t<0>::value_type ret = contrib;
    if constexpr(order >= 1)
      {
	ret += prev().calc_loc_potential(r);
      }

    return ret;
  }
#endif
  template <typename shift_t>
  Vector<typename get_t<0>::value_type, 3> calc_force(Vector<shift_t, 3> r) const
  {
    Vector<T, 3> F = {0,0,0};
    
    for (auto& i : indices)
      {
	F += (*this)[i] * i.factor
	  * Vector<T, 3>{ i.e[0] * pow(-r[0], i.e[0] - 1) * pow(-r[1], i.e[1]) * pow(-r[2], i.e[2]),
			  i.e[1] * pow(-r[0], i.e[0]) * pow(-r[1], i.e[1] - 1) * pow(-r[2], i.e[2]),
			  i.e[2] * pow(-r[0], i.e[0]) * pow(-r[1], i.e[1]) * pow(-r[2], i.e[2] - 1) };
      }

    Vector<typename get_t<0>::value_type, 3> ret = F.template cast<typename get_t<0>::value_type>();
    if constexpr(order >= 1)
      {
	ret += prev().calc_force(r);
      }
    return ret;
  }

  template<ostream_type STREAM>
  friend STREAM& operator<<(STREAM& s, const multipole& m)
  {
    s << "{";
    if constexpr (order >= 1)
      {
	s << static_cast<const prev_t&>(m) << ", ";
      }
    s << m.A << "}";
    return s;
  }
};

template<typename T, typename... TPREV>
array<Tint, pow<multipole<T, TPREV...>::order, 1>(3)> multipole<T, TPREV...>::indexmap;

GOOPAX_PREPARE_STRUCT(multipole)

template<typename T, typename... TPREV>
const array<typename multipole<T, TPREV...>::index_t, (multipole<T, TPREV...>::order + 1) * (multipole<T, TPREV...>::order + 2) / 2> multipole<T, TPREV...>::indices = []() {
  array<index_t, (order + 1) * (order + 2) / 2> ret;

  vector<index_t> sum;

  for (size_t k = 0; k < pow<order, 1>(size_t(3)); ++k)
    {
      size_t k2 = k;
      array<Tint, 3> cur = { 0, 0, 0 };
      array<Tint, order> indices;
      ranges::fill(indices, 0);
      for (uint l = 0; l < order; ++l)
        {
	  indices[l] = k2 % 3;
	  ++cur[k2 % 3];
	  k2 /= 3;
        }

      Tint ref;
      for (auto& s : sum)
        {
	  if (s.e == cur)
            {
	      ref = s.pos;
	      ++s.factor;
	      goto found_it;
            }
        }
      ref = sum.size();
      sum.push_back({ .pos = (int)sum.size(), .e = cur, .i = indices, .factor = 1 });
    found_it:
      indexmap[k] = ref;
    }

  assert(sum.size() == ret.size());

  auto rp = ret.begin();
  for (auto& s : sum)
    {
      *rp++ = s;
    }
  assert(rp == ret.end());
  return ret;
 }();


